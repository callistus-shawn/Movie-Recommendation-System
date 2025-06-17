from flask import Flask, request, render_template
import mysql.connector
app = Flask(__name__)
l=[]
from content import mov
import pickle
model=pickle.load(open("model.pkl","rb"))


@app.route("/")
def main():
    return render_template("index.html")

@app.route("/reg", methods=['post'])
def reg():
    name=request.form["name"]
    password=request.form["password"]

    conn=mysql.connector.connect(user="root",host="localhost",password="snucc@123",database="cia")
    cursor=conn.cursor()     
    if conn.is_connected():
           print("Connected")
    cursor.execute("INSERT INTO users (name,password)"
                   "values(%s,%s)",(str(name),str(password)))
    conn.commit()
    return render_template("pred.html")
    

@app.route("/login",methods=['post'])
def login():
    nam=request.form["name"]
    pas=request.form["password"]

    conn=mysql.connector.connect(user="root",host="localhost",password="snucc@123",database="cia")
    cursor=conn.cursor()     
    if conn.is_connected():
           print("Connected")
    
    cursor.execute('SELECT * FROM users WHERE name=%s and password=%s',(nam,pas))
    print("Done")
    rows=cursor.fetchone()
    if rows:
     return render_template("pred.html")
    else:
        error="Invalid Username or Password"
        return render_template("success.html",error=error)



@app.route("/predict",methods=["post"])
def predict():
    movie=request.form["movie"]
    print(movie)
    try:
     l=model.rec(movie)
     print(l)
     return render_template("pred.html",users=l)
    except:
     return render_template("pred.html",err="Check the spelling or try another movie")
    
    
    
    

@app.route("/register")
def register():
    return render_template("success.html")
    

if __name__=='__main__':
    app.run(host='localhost',port=5000, debug=True)