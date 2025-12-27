' Create Desktop Shortcut for Trail Camera Organizer
' Double-click this file after building the app

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get paths
strCurrentDir = fso.GetParentFolderName(WScript.ScriptFullName)
strDesktop = WshShell.SpecialFolders("Desktop")
strExePath = strCurrentDir & "\dist\TrailCamOrganizer\TrailCamOrganizer.exe"
strIconPath = strCurrentDir & "\icon.ico"

' Check if exe exists
If Not fso.FileExists(strExePath) Then
    MsgBox "TrailCamOrganizer.exe not found!" & vbCrLf & vbCrLf & _
           "Expected location:" & vbCrLf & strExePath & vbCrLf & vbCrLf & _
           "Please run BUILD.bat first to create the executable.", _
           vbExclamation, "Trail Camera Organizer"
    WScript.Quit
End If

' Create shortcut
Set oShortcut = WshShell.CreateShortcut(strDesktop & "\Trail Camera Organizer.lnk")
oShortcut.TargetPath = strExePath
oShortcut.WorkingDirectory = strCurrentDir & "\dist\TrailCamOrganizer"
oShortcut.Description = "Trail Camera Photo Organizer"

' Set icon if exists
If fso.FileExists(strIconPath) Then
    oShortcut.IconLocation = strIconPath
End If

oShortcut.Save

MsgBox "Desktop shortcut created successfully!" & vbCrLf & vbCrLf & _
       "Look for 'Trail Camera Organizer' on your desktop.", _
       vbInformation, "Trail Camera Organizer"
