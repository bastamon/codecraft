# coding=utf-8
import sys
import os
import predictor


# sys.argv[1]='../../TrainData_2015.1.1_2015.2.19.txt'#
# sys.argv[2]='../../inputczj.txt'#
# sys.argv[3]='../../output.txt'#
def main():
    print 'main function begin.'
    # if len(sys.argv) != 4:
    #     print 'parameter is incorrect!'
    #     print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
    #     exit(1)
    # Read the input files
    ecsDataPath = '../../data_2015_3.txt'#sys.argv[1]
    inputFilePath = '../../inputpar.txt'#sys.argv[2]
    resultFilePath = '../../output.txt'#sys.argv[3]

    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    # implementation the function predictVm
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
    # write the result to output file
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
    print 'main function end.'


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                if line != '\r\n' and line !='\n':
                    array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()
