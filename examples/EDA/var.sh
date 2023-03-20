if [ -z ${CALC+x} ]; then
  echo  "CALC not defined"
  exit
fi

VAR_SOURCED=""
PREFIX="LiNbO3"
INPUT_FILE="$PREFIX.$CALC.in"
OUTPUT_FILE="$PREFIX.$CALC.out"
