# ---------------------------------------------------------------------------
# hybridrt_apply_warnings(target)
#   Applies the project warning set PRIVATE to an own target.
# ---------------------------------------------------------------------------

function(hybridrt_apply_warnings target)
    if(MSVC)
        set(_host_flags /utf-8 /W4)
        set(_host_errors /WX)
        set(_cuda_flags -Xcompiler=/utf-8,/W4)
    else()
        set(_host_flags
            -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
            -Woverloaded-virtual -Wconversion -Wsign-conversion -Wimplicit-fallthrough)
        set(_host_errors -Werror)
        set(_cuda_flags -Xcompiler=-Wall,-Wextra)
    endif()
    # nvcc passes host compiler warnings through -Xcompiler and fails on its own warnings.
    set(_cuda $<COMPILE_LANGUAGE:CUDA>)
    set(_errors $<BOOL:${HYBRIDRT_WARNINGS_AS_ERRORS}>)
    target_compile_options(${target} PRIVATE
        "$<$<NOT:${_cuda}>:${_host_flags}>"
        "$<$<AND:$<NOT:${_cuda}>,${_errors}>:${_host_errors}>"
        "$<${_cuda}:${_cuda_flags}>"
        "$<$<AND:${_cuda},${_errors}>:-Werror=all-warnings>")
endfunction()
