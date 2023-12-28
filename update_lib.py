import os
import pdb
import re
import subprocess
import sys


def fetch_and_place(str1, str2=None):
    """checkout file str1 from remote repository path, then copy the file to str2, creating folder if needed.
    
    `str1`, `str2` are both filenames including extension.
    if `str2` is not provided, it is generated adding a `prefix` folder to `str1` removed root
    """
    
    prefix = "pyXsurf"
    
    # Perform git checkout
    git_checkout_command = ["git", "checkout", "upstream/master", "--", str1]
    try:
        subprocess.run(git_checkout_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during git checkout: {e}")
        return

    if str2 is None:
        # str2 = os.path.join(get_top_level_directory(str2),)
        str2 = os.path.join(prefix, *(str1.split(os.path.sep)[1:]))
    print(f"moving to {str2}")
    
    #pdb.set_trace()

    # Perform move operation
    os.makedirs(os.path.dirname(str2),exist_ok=True)
    try:
        os.rename(str1, str2)
    except OSError as e:
        print(f"Error moving file: {e}")
        return

    print(f"File {str1} successfully checked out and moved to {str2}")

def extract_and_format(input_string):
    # Regular expression to capture module and function names
    pattern = r"'(\w+)' from '([\w\.]+)'"
    match = re.search(pattern, input_string)

    if match:
        function_name, module_name = match.groups()
        module_path = module_name.replace('.', '\\')
        return f"{module_path}\\{function_name}.py"
    else:
        return None #"Pattern not found"
    
def extract_module_path(traceback_str):
    
    r"""
    Regular expression to find the module name in the ModuleNotFoundError
    
    From:
            'Traceback (most recent call last):
        File "C:\Users\kovor\Documents\python\pyXsurf\docs\source\examples\rotate_and_diff.py", line 20, in <module>
            from pySurf.data2D_class import Data2D
        File "C:\Users\kovor\Documents\python\pyXsurf\pyXsurf\pySurf\data2D_class.py", line 24, in <module>
            from pySurf.readers.format_reader import auto_reader
        ModuleNotFoundError: No module named 'pySurf.readers'
        
    Returns 'pySurf\readers\format_reader.py'
    """
    
    pattern = r"ModuleNotFoundError: No module named '([\w\.]+)'"
    match = re.search(pattern, traceback_str)

    if match:
        module_name = match.group(1)  # Extract the module name
        # Replace dots with backslashes and append '.py'
        module_path = module_name.replace('.', '\\') + '.py'
        return module_path
    else:
        return "Module not found in traceback"
    
def extract_failed_module(errstr):
    """
    Extracts the name of the failed module from an error string.
    
    :param errstr: The `errstr` string that represents an error message
    :return: the name of the failed module, extracted from the error string by searching a target string .and converted to path.
    """
    
    modtarget = "No module named"
    libtarget = "ImportError: cannot import name"
    
    i = errstr.find(modtarget)
    #pdb.set_trace()
    if i != -1: # find modtarget
        modname = extract_module_path(errstr)
        #pdb.set_trace()
        print(f'Extracted module {modname} as module')
        #
        # modname = errstr[i+len(modtarget):].strip()
        # modname = modname.replace("'", "")  #ugly
        # modname = modname.replace('.',os.path.sep)+'.py'    
        
    else: # check libtarget
        
        # string "No module named" not found 
        ## look for package (folder) error
        i = errstr.find(libtarget)
        modname = errstr[i+len(libtarget):].strip()
        modname = extract_and_format(modname)
        if modname is None:
            print("failed to extract modname, set to None.")        
        else:
            print(f'Extracted module {modname} as libtarget')
        
    return modname

def run_cmd(cmnd):
    """
    Run a shell command, returning output string if successful, extracted module if the error can be interpreted, or None otherwise.
    """    
    try:
        output = subprocess.check_output(
            cmnd, stderr=subprocess.STDOUT, shell=True, timeout=10,
            universal_newlines=True)
        
        # stdout = subprocess.run(['cat', '/tmp/text.txt'], check=True, capture_output=True, text=True).stdout
        
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL") #, exc.returncode, '\nO:\n', exc.output)
        output = extract_failed_module(exc.output)
        
        # if len(output) == 0:
        #     output = None
        #print('Extracted:\n',output)
        #pdb.set_trace()
        print('------------------------------------------')
    else:
        print("Output returned!")
        
    return output    

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <str1> <str2>")
    #     if len(sys.argv) == 1:  # raffazzonato per test wrapper
    #         test_capture_stderr()

    # else:
    if len(sys.argv)>=2:
        str1 = sys.argv[1]
        try:
            str2 = sys.argv[2]
        except IndexError:
            str2 = None
        fetch_and_place(str1, str2)

    # pyfile = r"docs\source\examples\rotate_and_diff.py"  # file to execute
    # result = subprocess.run(["python", pyfile], stderr=subprocess.PIPE)
    # print(">>>", result.stderr.decode(), "<<<")

    else:
        cmnd = r"python docs\source\examples\rotate_and_diff.py"
        #cmnd = r"python simple.py"
        maxiter = 3

        res = run_cmd(cmnd) 
        i=0
        #pdb.set_trace()
        while res and i<maxiter:
            if len(res):
                #pdb.set_trace()
                basename = os.path.join('source',res)
                fn = os.path.join(basename)
                fetch_and_place(fn)
            else:
                # check if it is module, must have been manually created.
                if os.path.exists(basename):
                    print (basename,"exists.")
                else:
                    print (basename,"doesn't exist.")
                    
            res = run_cmd(cmnd)
            i = i+1
        
        