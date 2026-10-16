// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/tools/run_tool.hpp
// Description: Runs a built command-line program and captures its exit code
//              and output for black-box tests.
// -----------------------------------------------------------------------------

#ifndef HRT_TOOLS_RUN_TOOL_HPP
#define HRT_TOOLS_RUN_TOOL_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/os/filesys.hpp>

#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string_view>
#include <utility>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace hrt::test {

struct ToolResult {
  int exit_code = -1;
  // stdout and stderr together.
  String output;
};

// Runs `program` with `args`, each passed as one quoted argument, writing its
// output to `log`.
inline auto run_tool(std::string_view program, std::initializer_list<String> args, const std::filesystem::path& log)
    -> ToolResult {
  // Appending instead of `"literal" + String` avoids a GCC 12 -Wrestrict false positive (PR105329).
  const auto append_quoted = [](String& command, std::string_view text) {
    command += '"';
    command += text;
    command += '"';
  };
  String command;
  append_quoted(command, program);
  for (const auto& arg : args) {
    command += ' ';
    append_quoted(command, arg);
  }
  command += " > ";
  append_quoted(command, log.string());
  command += " 2>&1";
#ifdef _WIN32
  // cmd.exe strips the outer quotes of a command line that starts with one.
  String wrapped;
  append_quoted(wrapped, command);
  command = std::move(wrapped);
#endif
  const auto status = std::system(command.c_str());
  ToolResult result;
#ifdef _WIN32
  result.exit_code = status;
#else
  result.exit_code = status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  result.output = kitzoo::os::read_file(log);
  return result;
}

} // namespace hrt::test

#endif // HRT_TOOLS_RUN_TOOL_HPP
