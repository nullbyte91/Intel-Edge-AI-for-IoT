from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-q_type", "--quantization_type", required=True, type=str,
                        help="There are two type quantization uint8 or int8 (uint8/int8)")
    
    parser.add_argument("-oldMax", "--old_Max", required=True, type=int,
                        help="Max value before quantization")

    parser.add_argument("-oldMin", "--old_Min", required=True, type=int,
                        help="Min value before quantization")

    parser.add_argument("-oldValue", "--old_Value", required=True, type=int,
                        help="Quantize old value")
    return parser

def main():
    args = build_argparser().parse_args()
    
    if args.quantization_type == "uint8":
        newMin= 0
        newMax = 255
    else:
        newMin = -128
        newMax = 127
    
    old_Range = args.old_Max - args.old_Min

    new_Range = newMax - newMin

    new_Value = ((args.old_Value - (args.old_Min)) * new_Range / old_Range) + newMin

    print("New Quantized Value: {}".format(new_Value))

if __name__ == "__main__":
    main()