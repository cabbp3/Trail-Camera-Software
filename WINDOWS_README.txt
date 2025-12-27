============================================
TRAIL CAMERA ORGANIZER - WINDOWS SETUP
============================================

FIRST TIME SETUP:
-----------------

1. Make sure Python 3.8 or newer is installed
   - Download from: https://www.python.org/downloads/
   - During install, CHECK the box "Add Python to PATH"

2. Double-click BUILD.bat
   - This will install dependencies and create the executable
   - Takes about 5-10 minutes
   - When done, it will ask if you want a desktop shortcut

3. That's it! Run the app from the desktop shortcut or RUN.bat


FILES IN THIS FOLDER:
---------------------

BUILD.bat              - Run this once to build the executable
RUN.bat                - Quick way to start the app after building
Create Desktop Shortcut.vbs - Creates a desktop shortcut (also in BUILD.bat)
icon.ico               - Windows icon for the app


AFTER BUILDING:
---------------

The built app will be in: dist\TrailCamOrganizer\TrailCamOrganizer.exe

You can:
- Use the desktop shortcut
- Double-click RUN.bat
- Navigate to dist\TrailCamOrganizer and run TrailCamOrganizer.exe


TROUBLESHOOTING:
----------------

"Python is not recognized..."
- Reinstall Python and make sure to check "Add Python to PATH"

Build fails with errors:
- Make sure you have internet connection (for downloading packages)
- Try running as Administrator

App won't start:
- Check if antivirus is blocking it
- Right-click the .exe and select "Run as administrator"


DATA LOCATIONS:
---------------

Your photos will be stored in: C:\TrailCamLibrary\
Database file: C:\Users\[YourName]\.trailcam\trailcam.db


============================================
