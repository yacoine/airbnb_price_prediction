import PySimpleGUI as sg
# All the stuff inside your window. 
layout = [  [sg.Text('Input the values in order to predict airbnb listings in NYC')],
            [sg.Text('Minimum nights per stay'), sg.InputText()],
            [sg.Text('Number of reviews'), sg.InputText()],
            [sg.Text('Number of listings, inclusive:'), sg.InputText()],
            [sg.Text('Select the appropriate neighbourhood')],
            [sg.Checkbox('Manhattan', default=True), sg.Checkbox('Brooklyn'), sg.Checkbox('Brooklyn'), sg.Checkbox('Bronx'),sg.Checkbox('Queens'),sg.Checkbox('Staten Island')],
            [sg.Button('Submit'), sg.Cancel()]]


layout1=[[sg.Button('Submit'), sg.Cancel()]]

# TO DO: INPUT VALUES for price prediction 
"""
minimum_night=input("Minimum nights")
number_of_reviews=input("Number of reviews")
calculated_host_listings_count=input("How many listings do you have, inclusive of this one.")
availability_365=input("How many days available per year")

neighbourhood_group=input("Neighbourhood")
#Give 5 options for the neighbourhood in button format
o Manhattan, Brooklyn, Bronx, Queens, Staten Island
All values will be equal to 0, unless the button is selected than the value is =1

room_type=input("Room type (1=Entire home/apt, 2=Private room, 3=Share room ")
#Give 3 options for the room type in button format
o Entire home/apt, Private room, Shared room
All values will be equal to 0, unless the button is selected than the value is =1
"""



# Create the Window
window = sg.Window('Get filename example', layout1)
# Event Loop to process "events" and get the "values" of the inputs
while True:             
    event, values = window.Read()
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        break

window.close()