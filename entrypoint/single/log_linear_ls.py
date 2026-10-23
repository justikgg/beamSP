from entrypoint._common import run_method


def main():
    run_method("Степенная: ограниченный Log-Linear Least Squares",
               model_names=["log_linear_ls"])


if __name__ == "__main__":
    main()
