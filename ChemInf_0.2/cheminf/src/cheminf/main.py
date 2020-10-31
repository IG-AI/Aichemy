from cheminf.utils import Timer, ChemInfOperator


@Timer
def main():
    """Run the main program."""
    cheminf = ChemInfOperator()

    if cheminf.mode == 'validate':
        cheminf.model.validate()
    elif cheminf.mode == 'build':
        cheminf.model.build()
    elif cheminf.mode == 'predict':
        cheminf.model.predict()
    elif cheminf.mode == 'improve':
        cheminf.model.improve()
    elif cheminf.mode == 'utils':
        cheminf.utils.run()

    print(cheminf.__module__)


if __name__ == '__main__':
    main()
