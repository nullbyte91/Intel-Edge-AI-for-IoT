#!/bin/bash
set -e

if [ -f "/opt/intel/openvino/bin/setupvars.sh" ]; then
    # Setup OpenVINO server
    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
fi

exec "$@"
