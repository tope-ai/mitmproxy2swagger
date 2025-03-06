"""
Changes in mitmprox2swagger.py:
1. Add import: from topeai_utils import is_param
2. Delete def is_param (in line 375)
3. Add the parameter "args" at the calls for is_param (in line 386): .. any(is_param(segment, args) for segment in segments)
4. Change the parameter "param_id = 0" to "param_counters = {"id": 0, "date": 0, "uuid": 0}" (in line 390)
4. Override the for in line 391 with this for:
    for segment in segments:
        param_type = is_param(segment, args)
        if param_type:
            param_count = param_counters[param_type]
            param_name = f"{param_type}{param_count if param_count > 1 else ''}"
            new_segments.append(f"{{{param_name}}}")
            param_counters[param_type] += 1
        else:
            new_segments.append(segment)
"""

import pandas as pd
from uuid import UUID

def is_param(param_value, args):
    # check if the parameter value matches one of the types: id, date or UUID.
    if args.param_regex.match(param_value):
        #print("id " + param_value)
        return "id"

    try:
        if pd.to_datetime(param_value, errors='coerce') is not pd.NaT:
            #print("datetime " + param_value)
            return "datetime"
    except ValueError:
        pass

    try:
        UUID(str(param_value))
        #print("UUID " + param_value)
        return "uuid"
    except ValueError:
        pass

    return None