# On Kubrick: conda activate tmp

import cs_lib as cs
import argparse

def main():

    parser = argparse.ArgumentParser(description = "Code for computing Rortex from WRF data")
    parser.add_argument('-i', '--input', type=str, help="Input file with tensor components", required=True)
    args = parser.parse_args()

    cs.rortex(args.input)

if __name__ == "__main__":
    main()
