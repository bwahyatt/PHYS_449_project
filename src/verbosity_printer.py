class VerbosityPrinter():
    def __init__(self, system_verbosity: int = 0):
        self.system_verbosity = system_verbosity
        
    def vprint(self, print_msg: str, msg_verbosity: int = 2) -> None:
        '''
        Prints a message if the message verbosity is smaller than or equal to the system verbosity

        Args:
            print_msg (str): The message to print
            msg_verbosity (int, optional): The verbosity level of the message. Defaults to 0.
        '''
        if msg_verbosity <= self.system_verbosity:
            print(print_msg)