if [ -z ${CALC+x} ]; then
  echo  "CALC not defined"
  exit
fi

VAR_SOURCED=""
PREFIX="MgO"
INPUT_FILE="$PREFIX.$CALC.in"
OUTPUT_FILE="$PREFIX.$CALC.out"
