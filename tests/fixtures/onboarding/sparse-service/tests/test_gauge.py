from gauge import main


def test_status():
    assert main()["status"] == "ok"
