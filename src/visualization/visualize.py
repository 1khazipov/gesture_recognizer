import torch
import matplotlib.pyplot as plt
import argparse

def show_plot_scores(model_name, train_scores, test_scores):
    """
    Display a plot of scores.

    Args:
        model_name (str): Model name
        train_scores (list): List of train scores to be plotted.
        test_scores (list): List of test scores to be plotted.
    """
    plt.figure()
    plt.plot(train_scores, label='Train score')
    plt.plot(test_scores, label='Test score')
    plt.title("F1 Score")

    plt.legend(loc="best")
    plt.title(f'Scores of {model_name} on epoch {len(test_scores)}')
    plt.show()
    
def show_plot_losses(model_name, train_losses, test_losses):
    """
    Display a plot of losses.

    Args:
        model_name (str): Model name
        train_losses (list): List of train losses to be plotted.
        test_losses (list): List of test losses to be plotted.
    """
    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.title("Loss")

    plt.legend(loc="best")
    plt.title(f'Losses of {model_name} on epoch {len(test_losses)}')
    plt.show()


def main(args):
    """
    Main function for loading a checkpoint and displaying validation scores.

    Args:
        args (argparse.Namespace): Command line arguments parsed using argparse.
    """
    load_ckpt_path = f'models/{args.model_name}'
    model_ckpt_path = load_ckpt_path + '/' + args.checkpoint_name

    checkpoint = torch.load(model_ckpt_path)

    train_scores = checkpoint['train_scores']
    test_scores = checkpoint['val_scores']
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['val_losses']

    show_plot_scores(args.model_name, train_scores, test_scores)
    show_plot_losses(args.model_name, train_losses, test_losses)

if __name__ == "__main__":
    checkpoint_name = 'best.pt'
    model_name = 'simple_lstm'

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", nargs='?', const=checkpoint_name, type=str, default=checkpoint_name, help='Name of the checkpoint to get results from')
    parser.add_argument('--model_name', nargs='?', const=model_name, type=str, default=model_name, help='Name of the model to use')

    args = parser.parse_args()

    main(args)