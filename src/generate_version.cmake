set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
# The committed state of the shader translator and the program cache: builds that share it keep
# one store of compiled programs (empty while either directory has local changes).
set(KYTY_GIT_SHADER_TREE "")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
		# The translator and the program cache's own files: the bindings code beside them in
		# renderer/pipeline (descriptors, heaps, blend mapping, barriers) does not change programs.
		set(PIPELINE_DIR src/graphics/host_gpu/renderer/pipeline)
		set(SHADER_DIRS src/graphics/shader ${PIPELINE_DIR}/pipelineCache.cpp ${PIPELINE_DIR}/pipelineCache.h
		    ${PIPELINE_DIR}/pipelineRecord.cpp ${PIPELINE_DIR}/pipelineRecord.h ${PIPELINE_DIR}/programStore.h
		    ${PIPELINE_DIR}/resourceMemo.h ${PIPELINE_DIR}/shaders.cpp)
		set(SHADER_TREE_ARGS "")
		foreach(path ${SHADER_DIRS})
			list(APPEND SHADER_TREE_ARGS "HEAD:${path}")
		endforeach()
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" rev-parse ${SHADER_TREE_ARGS}
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			OUTPUT_VARIABLE SHADER_TREES
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE SHADER_TREES_RESULT
			ERROR_QUIET
		)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD -- ${SHADER_DIRS}
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE SHADER_DIRTY_RESULT
			ERROR_QUIET
		)
		if(SHADER_TREES_RESULT EQUAL 0 AND SHADER_DIRTY_RESULT EQUAL 0)
			string(SHA1 KYTY_GIT_SHADER_TREE "${SHADER_TREES}")
		endif()
	endif()
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
