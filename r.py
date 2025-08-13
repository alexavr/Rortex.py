# On Kubrick: conda activate tmp

import cs_lib as cs
import sys

OUTPUT_PATH = f"./data_out/"

def main():

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        exit("No arguments provided.")

    cs.rortex(OUTPUT_PATH,filename)

if __name__ == "__main__":
    main()
