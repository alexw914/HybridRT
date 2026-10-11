# ---------------------------------------------------------------------------
# hybridrt_apply_warnings(target)
#   Applies the project warning set PRIVATE to an own target.
# ---------------------------------------------------------------------------

function(hybridrt_apply_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /W4
            $<$<BOOL:${HYBRIDRT_WARNINGS_AS_ERRORS}>:/WX>)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
            -Woverloaded-virtual -Wconversion -Wsign-conversion -Wimplicit-fallthrough
            $<$<BOOL:${HYBRIDRT_WARNINGS_AS_ERRORS}>:-Werror>)
    endif()
endfunction()
