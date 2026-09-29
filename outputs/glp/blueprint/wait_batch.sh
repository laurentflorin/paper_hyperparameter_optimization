#!/usr/bin/env bash
# Exit (one notification) when the 18-variable batch finishes, errors, or its queue ends.
cd "$(dirname "$0")/../../../logs"
LOG=canonical_large_nr_ladder.log
while true; do
  sleep 60
  if tr '\r' '\n' < $LOG | grep -q "All requested GLP stages completed"; then echo "finished: $LOG"; exit 0; fi
  if tr '\r' '\n' < $LOG | grep -qE "Traceback|Error:"; then echo "ERROR in $LOG (the queue retries automatically)"; exit 1; fi
  if ! kill -0 $(cat queue_large_nr.pid) 2>/dev/null; then echo "large queue ended"; exit 0; fi
done
