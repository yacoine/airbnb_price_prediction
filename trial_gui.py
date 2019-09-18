import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
import airbnb_predict

ui_name= 'trial.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(ui_name)


#minimum_night=''

class MyApp(QMainWindow):
	minimum_night=''
	number_reviews=''
	current_number_listings=''
	availability_365=''
	neigh=''
	room_type=''
	lower_limit=''
	upper_limit=''

	def __init__(self):
		super(MyApp, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.pushButton.clicked.connect(self.values_stored)
		#print(self.ui.minimum_nights)
        

	def values_stored(self):
		
		minimum_night=int(self.ui.minimum_nights.text())
		number_reviews=int(self.ui.number_reviews.text())
		current_number_listings=int(self.ui.current_number_listings.text())
		availability_365=int(self.ui.availability_365.text())
		neigh=self.ui.neighbourhood_val.currentIndex()
		room_type=self.ui.room_type.currentIndex()
		lower_limit=int(self.ui.lower_limit.value())
		upper_limit=int(self.ui.upper_limit.value())




		print("num rev.",number_reviews ," min night:" ,minimum_night)
		print('current listing:', current_number_listings, ' avail. ', availability_365)

		print('neighbourhood_group= ', neigh)

		print( 'lower:', lower_limit, ' higher: ', upper_limit)

		return 'works!'


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyApp()
	window.show()

	#airbnb_predict.trial_func()



	sys.exit(app.exec_())

	