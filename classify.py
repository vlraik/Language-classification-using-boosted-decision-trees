from utils import train, predict

def main():
    """
    The main function to get the user input.
    """
    train('train.dat','out','dt')
    predict('test.dat','out')
    while(True):
        command=input("Enter the option you want to use, type exit to quit: ").split()
        if command[0] == 'exit':
            exit()
        elif command[0] == 'train':
            examples = command[1]
            hypothesisOut = command[2]
            learning_type = command[3]
            print("Starting training...")
            train(examples, hypothesisOut, learning_type)
            print("Training is done.")

        elif command[0] == 'predict':
            hypothesis = command[1]
            file = command[2]
            print("Starting to predict...")
            print("The predictions are: ")
            predictions = predict(hypothesis, file)


if __name__ == '__main__':
    main()
