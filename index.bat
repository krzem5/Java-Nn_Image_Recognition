echo off
echo NUL>_.class&&del /s /f /q *.class
cls
javac -encoding utf8 com/krzem/nn_image_recognition/Main.java&&java -Dfile.encoding=UTF8 com/krzem/nn_image_recognition/Main
start /min cmd /c "echo NUL>_.class&&del /s /f /q *.class"