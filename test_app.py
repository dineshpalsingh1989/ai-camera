"""Self-check for the pure logic: python3 test_app.py"""
import app

# line crossing: a step from left to right across a vertical line counts, a step beside it does not
a, b = (100, 0), (100, 200)
assert app.crossed((90, 100), (110, 100), a, b)
assert not app.crossed((90, 100), (95, 100), a, b)
assert not app.crossed((90, 300), (110, 300), a, b)  # passes below the end of the line
assert (app.side(a, b, (110, 100)) > 0) != (app.side(a, b, (90, 100)) > 0)

# tracker keeps the ID while a box moves a little, gives a new ID to a far-away box
t = app.Tracker()
(id1, _, new1), = t.update([("person", 0.9, 100, 100, 200, 300)])
(id2, _, new2), = t.update([("person", 0.9, 110, 100, 210, 300)])
assert new1 and not new2 and id1 == id2
(id3, _, new3), = t.update([("person", 0.9, 500, 100, 600, 300)])
assert new3 and id3 != id1

# settings validation clamps numbers and rejects bad shapes
app.settings = app.load_settings()
s = app.clean_settings({"confidence": 5, "cameras": [{"source": "0", "zones": [{"points": [[0, 0], [2, 0], [1, 1]]}]}]})
assert s["confidence"] == 0.95 and s["cameras"][0]["zones"][0]["points"][1] == [1, 0]
s = app.clean_settings({"cameras": [{"source": "0", "conn": {"type": "usb", "usb": "0", "evil": "x"}}]})
assert s["cameras"][0]["conn"] == {"type": "usb", "usb": "0"}
try:
    app.clean_settings({"cameras": [{"source": "0", "lines": [{"points": [[0, 0]]}]}]})
    raise AssertionError("one-point line accepted")
except ValueError:
    pass
assert app.clean_inspection({"votes": 9, "window": 3}, app.COCO)["votes"] == 3  # can't need more votes than frames
s = app.clean_settings({"custom_classes": ["Egg", "egg", " "], "alert_classes": ["egg", "unicorn"],
                        "cameras": [{"source": "0", "inspection": {"target": "egg"}}]})
assert s["custom_classes"] == ["egg"] and s["alert_classes"] == ["egg"] and s["cameras"][0]["inspection"]["target"] == "egg"

# anomaly model: trained on good eggs only, a defective egg must score higher than an unseen good one
try:
    import torch  # noqa: F401
    import make_demo_eggs as eggs
    good = [eggs.egg(i, None)[0].astype("uint8") for i in range(20)]
    model = app.AnomalyModel.fit(good, {}, (128, 128))
    (ok_score, _), (bad_score, _) = model.score([eggs.egg(99, None)[0].astype("uint8"), eggs.egg(99, "spots")[0].astype("uint8")])
    assert bad_score > ok_score * 1.2, (ok_score, bad_score)
    print(f"anomaly check: good {ok_score:.2f} vs defect {bad_score:.2f}")
except ImportError:
    print("anomaly check skipped (no PyTorch)")
print("all checks passed")
