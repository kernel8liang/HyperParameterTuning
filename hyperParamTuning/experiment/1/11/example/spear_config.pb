language: PYTHON
name: "keras_spearmint_experiment.cifar10_subclass_cnn"

variable {
        name: "lr"
        type: FLOAT
        size: 1
        min:  0.01
        max:  0.1
        }

variable {
        name: "is1"
        type: INT
        size: 1
        min:  3
        max:  9
        }
variable {
        name: "is2"
        type: INT
        size: 1
        min:  3
        max:  9
        }

variable {
        name: "is3"
        type: INT
        size: 1
        min:  3
        max:  9
        }
variable {
        name: "is4"
        type: INT
        size: 1
        min:  3
        max:  9
        }
variable {
        name: "ps1"
        type: INT
        size: 1
        min:  2
        max:  4
        }
variable {
        name: "ps2"
        type: INT
        size: 1
        min:  2
        max:  4
        }
variable {
        name: "mo"
        type: FLOAT
        size: 1
        min:  0.5
        max:  0.9
        }
