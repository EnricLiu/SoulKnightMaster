## 1280 x 720 

### Positions below are counted taking the top-left corner as (0, 0), while adb takes the bottom-left as (0, 0).
### No wonder why that the x and y reported by ADB are SWAPPED. ?????????
### A translation is needed. (x, y) => (DISPLAY_HEIGHT - y, x)

### joystick
- center: ( 240, 540) --> **( 240, 180)**
- click inside **radius=270-1** is accepted as joystick movement.

### attack btn
- center: (1020, 580) --> **(1020, 140)**
- btn radius: **89-1**.

### skill btn
- center: (1180, 600) --> **(1180, 120)**
- btn radius: **60-1**.

### switching btn
- center: (1180, 450) --> **(1180, 270)**
- btn radius: **60**.
- seems like not a circle or rectangle?? capsule like?? **confused**
