# Place SDK DLLs beside executables so Windows does not load a different
# ONNX Runtime from System32 before searching PATH.
function(hybridrt_deploy_runtime target)
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DHYBRIDRT_RUNTIME_DLLS=$<TARGET_RUNTIME_DLLS:${target}>"
                "-DHYBRIDRT_RUNTIME_DIR=$<TARGET_FILE_DIR:${target}>"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CopyRuntimeDlls.cmake"
            VERBATIM)
    endif()
endfunction()
