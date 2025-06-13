#!/usr/bin/env python3

import benchexec.util as util
import benchexec.tools.template
import benchexec.result as result

class Tool(benchexec.tools.template.BaseTool2):
    """
    Tool info for Z3 PolySAT.
    """

    def executable(self, tool_locator):
        return tool_locator.find_executable("z3-poly")

    def name(self):
        return "Z3-poly"

    def determine_result(self, run):
        """
        Parse the output of Z3 and extract the verification result.
        """
        for line in run.output:
            if line.strip() == "sat":
                return result.RESULT_TRUE_PROP
            elif line.strip() == "unsat":
                return result.RESULT_FALSE_PROP
            elif "timeout" in line.lower():
                return result.RESULT_TIMEOUT
            elif "unknown" in line.strip():
                return result.RESULT_UNKNOWN
        return result.RESULT_ERROR