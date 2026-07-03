# Ocular Gesture Modules (OGM)

**OGM** is a Python API (for now) that allows the implementation of highly customizable eye gestures.

Currently, OGM supports only blinking gestures. These are fully customizable, allowing the API user to build potentially infinite combinations of actions (RAM permitting).

---

## Installation
If you download this repository, navigate inside and run:

```bash
pip install .
```

Alternatively:
```bash
pip install ogm
```


---

## Usage

> [!WARNING]
> **OGM IS IN ACTIVE DEVELOPMENT**
> 
> It is highly likely that the way to do certain things with OGM might change frequently across versions, as the library is still in early development.
> To get an idea of what's coming, check out the roadmap below.

I have included DocStrings within the API files, so if any information is missing here, you should still have access to everything you need right in your IDE.

***Usage Example:***
```python
from ogm import ActionType, BlinkDetector, CameraConfig


blink_detector = BlinkDetector()

# Your callback function to handle actions
def on_actions(azioni: list[tuple[ActionType, int | None]]):
	match azioni:
		# If the user blinks the left eye
		case [(ActionType.LEFT, _)]:
			#
			# ANYTHING YOU WANT TO BE DONE
			#
			blink_detector.reset_log()

		# If the user blinks both eyes
		case [(ActionType.BOTH, _)]:
			#
			# ANYTHING YOU WANT TO BE DONE
			#
			blink_detector.reset_log()

		# Combo example: Right eye followed by Left eye (max pause 1000ms)
		case [*_, (ActionType.RIGHT, pause), (ActionType.LEFT, _)]:
			if pause is not None and pause <= 1000:
				#
				# ANYTHING YOU WANT TO BE DONE
				#
				blink_detector.reset_log()
		
		# Wait state: If the user blinks the right eye (waiting for combo)
		case [*_, (ActionType.RIGHT, _)]:
			#
			# ANYTHING YOU WANT TO BE DONE
			#
			pass

		# Any other sequence of actions is ignored
		case _:
			pass
		

if __name__ == "__main__":
	# Callback binding
	blink_detector.on_blink = on_actions
	
	# Configuration of the camera
	my_camera = CameraConfig()  # Remember to set it to 0 if you only have one camera!
	blink_detector.start(mode="detect", camera_config=my_camera)
	# ...
	# ...
	# Your code...
	# ...
	# ...
	blink_detector.close()
```

***You can also perform automatic calibration:***
```python
# ... 
# ...
# ...
# Your callback function for auto-calibration
def on_calibration(left_eye: float, right_eye: float):
	blink_detector.left_ear_threshold = left_eye
	blink_detector.right_ear_threshold = right_eye
	
if __name__ == "__main__":
	# Callback bindings
	blink_detector.on_blink = on_actions
	blink_detector.on_calibration_callback = on_calibration
	
	# Configuration of the camera
	my_camera = CameraConfig()  # Remember to set it to 0 if you only have one camera!
	blink_detector.start(mode="calibrate", camera_config=my_camera)
	# ...
	# ...
	# Your code...
	# ...
	# ...
	blink_detector.close()
```

---

## Acknowledgments & Legal
This library is built as a wrapper and mathematical layer on top of [Google MediaPipe](https://developers.google.com/mediapipe) for high-performance, real-time facial landmark detection. 

The OGM library bundles the `face_landmarker.task` model, which is provided by Google under the **Apache License 2.0**. 
For more details, please refer to the official [MediaPipe repository](https://github.com/google-ai-edge/mediapipe).

---

## Development Roadmap
The roadmap I have set for the development of this API is:
1. Blinking gestures module <-- **In Testing Phase**
2. Eye movement gestures module.
3. Eyebrow movement gestures module.
4. Hand movement gestures module.
5. Rewriting the core API in C++/Rust (yet to be decided).
6. Creating bindings for C++/Rust to other languages.

---

```
###
# project: Ocular Gesture Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
###
```

**To learn more about me, check <a href="https://valerioditommaso.dev/en">my website</a>.**
