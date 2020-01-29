import random as rd

def block_function(a_str):
	init_vector = rd.randint(2,2000)
	print("IV value {}".format(init_vector))

	pt =[]

	for i in a_str: #Breaking string apart per letter
		pt.append(i)

	for x in pt:
		print(ord(x),x) #Letter to ascii
		print(ord(x) ^ init_vector) #XORed with Initialization Vector


if __name__ == '__main__':
	a_str = input("Input string:")
	print(a_str)
	block_function(a_str)
