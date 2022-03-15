REM Compile and upload new version to TestPyPI. Version must be updated every time.

pipreqs .

python setup.py bdist_wheel

for /f %%i in ('dir /b/a-d/od/t:c dist') do set LAST=%%i

#echo Upload %LAST%

twine upload --repository testpypi -u vincenzooo  --skip-existing dist/%LAST%
rem dist/pyXsurf-0.1.25-py3-none-any.whl