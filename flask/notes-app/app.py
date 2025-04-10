from flask import Flask

app = Flask(__name__) #use the name of the file for the app name

@app.route("/") 
def hello(): 
  return "Hello world"

# if __name__ == "__main__": 
#   app.run(debug=True) 