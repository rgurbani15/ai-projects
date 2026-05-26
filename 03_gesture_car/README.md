\# Project 3: AI Hand-Controlled F1 Car



Control an on-screen F1 car using your hand position via webcam.



\## How It Works



\- Custom YOLOv8 hand detector trained on 4,800 EgoHands images

\- Hand position maps to car steering and speed

\- Real-time control at 30 FPS



\## Files



| File | Purpose |

|---|---|

| `gesture\_car.py` | Main game — hand tracking + car physics |

| `hand\_detector.pt` | Custom trained hand detection model |

| `hand\_tracker.py` | Standalone hand tracking demo |



\## Controls



| Hand Position | Action |

|---|---|

| Center | Stop |

| Left | Steer left |

| Right | Steer right |

| Up | Accelerate |

| Down | Reverse |



\## How to Run



```bash

python gesture\_car.py

