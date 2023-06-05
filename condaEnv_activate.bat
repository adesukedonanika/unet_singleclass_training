@REM %windir%\System32\cmd.exe "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat C:\Users\adesu\.conda\envs\gpugdal

call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\Users\adesu\.conda\envs\gpugdal


@REM GoogleDriveがマウントされるまで待つ
timeout 1

G:
cd H:\マイドライブ\Forest/src

jupyter lab
/k


@REM "C:\Users\adesu\.jupyter\jupyter_notebook_config.py"　に下記を追加すると他PCからJupyterを開ける
@REM c.NotebookApp.open_browser = False
@REM c.NotebookApp.ip = "*"
@REM c.NotebookApp.port = 8888


