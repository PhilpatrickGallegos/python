 #pgallegos
#j.pino
#10/22/2016
#I am writing program for chapter 10.
#I have spent about 10 hrs on this program.
#Honor Code: I pledge that this program represents my own program code.
#I recieved help from(no one) in designing and debugging my program.

# The employee class simulates the animal/pet
class Employee:

# The __init__ method initializes the employee data

    def __init__ (self, name="unknown", ID_number="unknown", department="unknown", job_title="unknown"):
        self.__name=name
        self.__ID_number=ID_number
        self.__department=department
        self.__job_title=job_title

    # The set_name accepts args for the employees name.

    def set_name(self, name):
        self.__name= name
    # The set_ID_number accepts args for the ID number of the employee.

    def set_ID_number(self, ID_number):
        self.__ID_number=ID_number
    # The set_age accepts args for the employees department.

    def set_department(self, department):
        self.__department=department
     # The set_job_tile accepts args for the employees job title.

    def set_job_title(self, job_title):
        self.__job_title=job_title
    # The set_info accepts args for the employees job info.

    def set_info(self, info):
        self.__info=info
        
    # The get_name returns the name of the employee.

    def get_name(self):
        return self.__name
    # The get_ID_number_type returns the ID number of the employee.

    def get_ID_number(self):
        return self.__ID_number
    # The get_department returns the department of the employee.

    def get_department(self):
        return self.__department
    
    # The get_job_title returns the employee's job title.
    def get_job_title(self):
        return self.__job_title
    
    # The get_info returns all the employee's info.
    def get_info(self):
        return self.__name,self.__ID_number,self.__department,self.__job_title
    
    
        

    
    
