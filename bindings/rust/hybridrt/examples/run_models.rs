// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/examples/run_models.rs
// Description: Runs one algo per task and OCR from an .hrtc package or a JSON
//              manifest on the test images and prints the results.
// -----------------------------------------------------------------------------

use std::path::Path;

use hybridrt::{Image, ImageFormat, Model, ModelOptions, OcrModel, Package, Results, Task};

/// (algo, image): one per task.
const CASES: [(&str, &str); 5] = [
    ("yolov8n_detect", "bus.jpg"),
    ("yolov8n_obb", "airport.jpg"),
    ("yolov8n_pose", "bus.jpg"),
    ("yolov8n_segment", "bus.jpg"),
    ("yolov8n_classify", "bus.jpg"),
];
const OCR: (&str, &str, &str) = ("ppocrv6_tiny_det", "ppocrv6_small_rec", "ocr_document.jpg");

/// RGB pixels, width and height of a JPEG file, or None when it cannot be read.
fn read_jpeg(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let jpeg = std::fs::read(path).ok()?;
    let mut decoder = zune_jpeg::JpegDecoder::new(&jpeg);
    let rgb = decoder.decode().ok()?;
    let (width, height) = decoder.dimensions()?;
    Some((rgb, width as u32, height as u32))
}

fn print_results(task: Task, results: &Results) {
    match task {
        Task::Classify => {
            let scores = results.scores(0).unwrap_or_default();
            let mut top: Vec<usize> = (0..scores.len()).collect();
            top.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let top: Vec<String> = top
                .iter()
                .take(5)
                .map(|&i| format!("class {i} {:.3}", scores[i]))
                .collect();
            println!("  top-5: {}", top.join(", "));
        }
        Task::Rotated => {
            for b in results.rotated_boxes() {
                println!(
                    "  class {} score {:.3} center ({:.1}, {:.1}) size {:.1}x{:.1} angle {:.1}",
                    b.class_id, b.score, b.center_x, b.center_y, b.width, b.height, b.angle
                );
            }
        }
        _ => {
            for (index, b) in results.boxes().iter().enumerate() {
                let keypoints = match results.keypoints(index) {
                    Some(points) if task == Task::Keypoint => format!(" keypoints {}", points.len()),
                    _ => String::new(),
                };
                println!(
                    "  class {} score {:.3} box [{:.1}, {:.1}, {:.1}, {:.1}]{keypoints}",
                    b.class_id, b.score, b.x, b.y, b.width, b.height
                );
            }
            if task == Task::Segment {
                println!("  masks: {}", (0..).map_while(|index| results.mask(index)).count());
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <model.hrtc|manifest.json> [images-dir]", args[0]);
        std::process::exit(1);
    }
    let images = Path::new(args.get(2).map_or("assets/test/images", String::as_str));
    let package = Package::load(&args[1])?;
    let algos = package.algos();
    let has = |algo: &str| algos.iter().any(|name| name == algo);

    for (algo, name) in CASES {
        let Some((rgb, width, height)) = read_jpeg(&images.join(name)).filter(|_| has(algo)) else {
            println!("{algo}: skipped, needs the algo and {name}");
            continue;
        };
        let image = Image::new(&rgb, ImageFormat::InterRgb, width, height)?;
        let model = Model::new(&package, Some(algo), ModelOptions::default())?;
        let results = model.run(&image, None)?;
        println!("{algo} ({:?}) on {name}: {} results", model.task(), results.len());
        print_results(model.task(), &results);
    }

    let (detection, recognition, name) = OCR;
    let Some((rgb, width, height)) = read_jpeg(&images.join(name)).filter(|_| has(detection) && has(recognition))
    else {
        println!("ocr: skipped, needs {detection}, {recognition} and {name}");
        return Ok(());
    };
    let image = Image::new(&rgb, ImageFormat::InterRgb, width, height)?;
    let mut model = OcrModel::new(&package, detection, recognition)?;
    let texts = model.run(&image)?;
    println!("ocr ({detection} + {recognition}) on {name}: {} texts", texts.len());
    for item in texts {
        println!(
            "  {:?} rect {:?} score {:.3}",
            item.text, item.rect, item.recognition_score
        );
    }
    Ok(())
}
