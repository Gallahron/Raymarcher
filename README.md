# Raymarcher
A bare-bones raymarch engine featuring small number of primitives. Programmed from scratch in C++, using NVIDIA Cuda for GPU parallelisation, and SDL2 to draw the graphics to the screen.
[Video Demo](https://www.youtube.com/watch?v=rqYLiHtZyNM)

Positive Features:
- Rendered my website's header image [link](http://www.joemo.co.uk)
- Draws 3D graphics to the screen
- Acceptable framerates at lower resolutions
  - Pixel skipping enables larger screen sizes with low resolutions

Negative Features:
- Framerate is unusably slow at desired framerates
  - Current workaround is to skip drawing a number of pixels. If the pixels don't line up in neat rows, the human eye can fill in the gaps fairly convincingly. This means that we can get a 100%+ improvement in FPS with a minor drop in clarity.
- Lighting produces dark spots at certain ray angles.

Lessons Learned:
- Use libraries for the purpose they're designed for.
  - CUDA is good for computational applications but kernel initialisation and memory copy times were too slow for enjoyable realtime use.
  - The program would likely have ran a lot faster if it had been written in an OpenGL shader.
- Implicit conversions take a lot of time.
  - Was able to get 100% FPS increase by removing implicit double -> float conversions.
  
  Possible Extensions:
  - Use semaphores to allow concurrent CPU and GPU processing.
    - Currently program crashes because they try to access the same memory.
  - Implement a persistent kernel
    - Avoids waiting for kernel initialisation each frame.
    - Enables CPU and GPU concurrency.
