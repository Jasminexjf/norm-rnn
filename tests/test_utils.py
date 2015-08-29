def test_progress_bar():
    from utils import TrainProgressBar
    progress_bar = TrainProgressBar(10, 10)
    progress_bar.fit_update(zip(range(10), range(10)))
    progress_bar.val_update(zip(range(5), range(5)))


test_progress_bar()