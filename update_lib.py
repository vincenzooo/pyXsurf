import os
import subprocess
import sys
import pdb


def get_result(function=None, *args, **kwargs):
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("vvvvvvvvvv")
    result = None if function is None else function(*args, **kwargs)
    sys.stdout = old_stdout
    s = mystdout.getvalue()

    sys.stderr.write(s)
    print("^^^^^^^^^^")

    return result


import sys
from functools import wraps
from io import StringIO


def capture_stderr(function):
    """
    a wrapper which capture stderr for any function.

    When the decorated function is called, it captures anything that function prints
    to stderr and redirects it to a StringIO object. After the function call, it restores the original stderr, writes the captured output to stderr, and prints your custom messages.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # Save the current stderr
        old_stderr = sys.stderr
        # Redirect stderr to a StringIO object
        sys.stderr = mystderr = StringIO()

        # Your custom behavior before the function call
        print("vvvvvvvvvv")

        # Call the function and capture the result
        try:
            result = function(*args, **kwargs)
        finally:
            # Restore the original stderr
            sys.stderr = old_stderr

        # Get the captured output
        s = mystderr.getvalue()

        if s:
            print("captured!")

        # Your custom behavior after the function call
        sys.stderr.write(s)
        print("^^^^^^^^^^")

        return result

    return wrapper


def test_capture_stderr():
    # Example usage
    @capture_stderr
    def example_function():
        print("This is a test.", corbezzoli)

    return example_function()


#@capture_stderr
def main(str1, str2=None):
    # Perform git checkout
    git_checkout_command = ["git", "checkout", "upstream/master", "--", str1]
    try:
        subprocess.run(git_checkout_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during git checkout: {e}")
        return

    if str2 is None:
        # str2 = os.path.join(get_top_level_directory(str2),)
        str2 = os.path.join("pyXsurf", *(str1.split(os.path.sep)[1:]))
    print(f"moving to {str2}", str2)
    
    #pdb.set_trace()

    # Perform move operation
    try:
        os.rename(str1, str2)
    except OSError as e:
        print(f"Error moving file: {e}")
        return

    print(f"File {str1} successfully checked out and moved to {str2}")

def extract_failed_module(errstr):
    """
    Extracts the name of the failed module from an error string.
    
    :param errstr: The `errstr` string that represents an error message
    :return: the name of the failed module, extracted from the error string by searching a target string .
    """
    
    target = "No module named"
    i = errstr.find(target)
    if i == -1: modname = ''
    else:
        modname = errstr[i+len(target):].strip()
        modname = modname.replace("'", "")  #ugly
    return modname
    

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <str1> <str2>")
    #     if len(sys.argv) == 1:  # raffazzonato per test wrapper
    #         test_capture_stderr()

    # else:
    #     str1 = sys.argv[1]
    #     try:
    #         str2 = sys.argv[2]
    #     except IndexError:
    #         str2 = None
    #     main(str1, str2)

    # pyfile = r"docs\source\examples\rotate_and_diff.py"  # file to execute
    # result = subprocess.run(["python", pyfile], stderr=subprocess.PIPE)
    # print(">>>", result.stderr.decode(), "<<<")

    def run_cmd(cmnd):
        """
        Run a shell command, returning output string if successful, extracted module if the error can be interpreted, or empty string otherwise.
        """
        import subprocess
        try:
            output = subprocess.check_output(
                cmnd, stderr=subprocess.STDOUT, shell=True, timeout=10,
                universal_newlines=True)
            
            # stdout = subprocess.run(['cat', '/tmp/text.txt'], check=True, capture_output=True, text=True).stdout
            
        except subprocess.CalledProcessError as exc:
            print("Status : FAIL") #, exc.returncode, '\nO:\n', exc.output)
            output = extract_failed_module(exc.output)
            print('Extracted:\n',output)
            #pdb.set_trace()
            print('------------------------------------------')
        else:
            print("Output returned!")
            
        return output

    cmnd = r"python docs\source\examples\rotate_and_diff.py"
    #cmnd = r"dir"

    res = run_cmd(cmnd)
    if len(res):
        #pdb.set_trace()
        fn = os.path.join('source',res.replace('.',os.path.sep)+'.py')
        main(fn)