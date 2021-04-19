import tensorflow as tf

class MetricTracker:
    """ Tracks the values of a single metric during training / validation using a floating average."""

    def __init__(self, name, string_format='{:.3f}'):
        """ Default initialisation function.

        @param name: Name of the tracked metric.
        @param string_format: Formatting string which is used to print the current state.
        """
        self.tracker = tf.keras.metrics.Mean()
        self.name = name
        self.string_format = string_format

    def update(self, value):
        """ Updates the tracked floating average using the specified value.

        @param value: Current metric value.
        """
        self.tracker.update_state(value)

    def reset(self):
        """ Resets the tracked floating average to its initial state. """
        self.tracker = tf.keras.metrics.Mean()

    def evaluate(self):
        """ Computes the floating average based on all metric values specified so far.

        @return: The current floating average.
        """
        return self.tracker.result()

    def to_string(self):
        return self.string_format.format(self.evaluate())


class MetricTrackerManager:
    """ Manages multiple metric tracker objects. """

    def __init__(self):
        """ Default initialisation function. """
        self.training_metrics = {}
        self.validation_metrics = {}

    def add_metric(self, name , format='{:.3f}', is_training=None):
        """ Adds a new metric to be tracked.

        @param name: Name of the tracked metric.
        @param format: Formatting string which is used to print the current state.
        @param is_training: Specifies if the metric is used for training or validation (or both if None is specified).
        """
        if is_training is None:
            self.training_metrics[name] = MetricTracker(name, format)
            self.validation_metrics[name] = MetricTracker(name, format)
        elif is_training:
            self.training_metrics[name] = MetricTracker(name, format)
        else:
            self.validation_metrics[name] = MetricTracker(name, format)

    def add_metrics(self, names, formats, is_training=None):
        """ Adds multiple metrics at once.

        @param names: List containing the metric names.
        @param formats: List containing the formatting strings used to print the current metric states.
        @param is_training: Specifies if the metric is used for training or validation (or both if None is specified).
        """
        for index in range(len(names)):
            self.add_metric(names[index], formats[index], is_training)

    def update(self, is_training, names, values):
        """ Updates one of the tracked metrics.

        @param is_training: Specifies if the metric is used for training or validation.
        @param names: Name of the tracked metric.
        @param values: Current metric value used to update the metric tracker.
        """
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics

        for index, key in enumerate(names):
            metrics[key].update(values[index])

    def reset(self):
        """ Resets all metric trackers to their initial states. """
        for key in self.training_metrics:
            self.training_metrics[key].reset()
        for key in self.validation_metrics:
            self.validation_metrics[key].reset()

    def get_all_data(self, is_training):
        """ Returns the current values of all tracked metrics.

        @param is_training: Specifies if the data is requested for training or validation metrics.
        @return: A dictionary containing the current values of all tracked metrics with the metric names as keys.
        """
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics
        data = {}
        for key in metrics:
            data[key] = metrics[key].evaluate()
        return data

    def get_data(self, metric_name, is_training):
        """ Returns the current value of a specific tracked metric.

        @param is_training: Specifies if the data is requested for a training or a validation metric.
        @return: The current value of the specified metric.
        """
        if is_training:
            return self.training_metrics[metric_name].evaluate()
        else:
            return self.validation_metrics[metric_name].evaluate()

    def to_string(self, is_training):
        """ Prints the current values of all tracked metrics to string using the specified formatting strings.

        @param is_training: Specifies if the data is requested for training or validation metrics.
        @return: A string consisting of the tracked metric names and values.
        """
        if is_training:
            result = 'Training:    '
            metrics = self.training_metrics
        else:
            result = 'Validation:  '
            metrics = self.validation_metrics

        for key in metrics:
            result += metrics[key].name + ': ' + metrics[key].to_string() + '   '
        return result
