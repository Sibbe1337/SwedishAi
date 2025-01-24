from database import SessionLocal, User, Base, engine
from datetime import datetime
from passlib.context import CryptContext
import sqlalchemy as sa

# Droppa alla tabeller med CASCADE
print("Återskapar databastabeller...")
with engine.connect() as conn:
    # Inaktivera foreign key checks temporärt
    conn.execute(sa.text("DROP SCHEMA public CASCADE"))
    conn.execute(sa.text("CREATE SCHEMA public"))
    conn.commit()

# Skapa tabeller på nytt
Base.metadata.create_all(bind=engine)

# Setup password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_test_user():
    db = SessionLocal()
    
    # Kolla om användaren redan finns
    if db.query(User).filter(User.email == "test@example.com").first():
        print("Testanvändare finns redan")
        return
    
    # Skapa ny användare
    test_user = User(
        email="test@example.com",
        hashed_password=pwd_context.hash("password123"),
        is_premium=False,
        message_count=0,
        last_reset=datetime.utcnow()
    )
    
    db.add(test_user)
    db.commit()
    print("Testanvändare skapad:")
    print("Email: test@example.com")
    print("Lösenord: password123")

if __name__ == "__main__":
    create_test_user() 