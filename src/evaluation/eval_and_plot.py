from conv import convolved_1d  # pip install conv
import matplotlib.pyplot as plt


def eval_and_plot(trained_pipelines, X_test, y_test, plot_range):
    all_plot_y_axis = dict()
    for (model_name, model) in trained_pipelines.items():
        plot_x_axis, plot_y_axis = eval_for_min_char_range(
            model, X_test, y_test, plot_range
        )
        all_plot_y_axis[model_name] = plot_y_axis
    plot_result(plot_x_axis, all_plot_y_axis)


def eval_for_min_char_range(trained_pipeline, X_test, y_test, plot_range=30):
    plot_x_axis = list(range(3, plot_range))
    plot_y_axis = []
    for char_shown in range(3, plot_range):

        convolved_X_test, convolved_y_test = convolve_test_set(X_test, y_test, kernel_size=char_shown)
        score = trained_pipeline.score(convolved_X_test, convolved_y_test)
        plot_y_axis.append(score * 100)

    return plot_x_axis, plot_y_axis


def convolve_test_set(X_test, y_test, kernel_size, stride=None):
    """
    Split chunks of text into chunks of maximum length of "kernel_size" chars each.
    """
    if stride is None:
        stride = kernel_size

    convolved_X_test = []
    convolved_y_test = []

    for x_full, y in zip(X_test, y_test):
        for x in convolved_1d(x_full, kernel_size=kernel_size, stride=stride, padding='VALID', default_value=' '):
            convolved_X_test.append("".join(x))
            convolved_y_test.append(y)

    return convolved_X_test, convolved_y_test


def plot_result(plot_x_axis, all_plot_y_axis):
    plt.figure(figsize=(12, 12))
    plt.title("Performance of the models")
    for (model_name, plot_y_axis) in all_plot_y_axis.items():
        plt.plot(plot_x_axis, plot_y_axis, label=model_name)
    plt.ylabel('Number of characters seen by model')
    plt.xlabel('Accuracy, in percents (%)')
    plt.legend()
    plt.show()
