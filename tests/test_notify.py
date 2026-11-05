from door_inspection.notify import TelegramNotifier


def test_without_token_prints_and_does_not_crash(capsys):
    tg = TelegramNotifier(token="", chat_id=None)
    tg.send("hello")
    assert "hello" in capsys.readouterr().out


def test_alerts_only_on_defect_classes():
    tg = TelegramNotifier(token="", alert_classes=["Rubber_missing"])
    assert tg.maybe_alert("Rubber_missing", None)
    assert not tg.maybe_alert("Door_closed", None)


def test_alert_cooldown_per_class():
    tg = TelegramNotifier(token="", alert_classes=["Door_Open", "Rubber_missing"], cooldown_s=60)
    assert tg.maybe_alert("Door_Open", None)
    assert not tg.maybe_alert("Door_Open", None)       # within cooldown
    assert tg.maybe_alert("Rubber_missing", None)      # different class, own timer
