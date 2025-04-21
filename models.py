from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import hashlib
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(64), nullable=False)
    
    def set_password(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    user_query = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    additional_context = Column(Text)
    feedback = relationship("Feedback", back_populates="chat_history", uselist=False)

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    chat_history_id = Column(Integer, ForeignKey('chat_history.id'), nullable=False)
    is_positive = Column(Integer, nullable=False)  # 1 for thumbs up, 0 for thumbs down
    feedback_text = Column(Text)  # Optional text feedback for thumbs down
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat_history = relationship("ChatHistory", back_populates="feedback")

# Database setup
DATABASE_URL = "sqlite:///chat_history.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    
    # Create default admin user if not exists
    db = SessionLocal()
    if not db.query(User).filter_by(username="admin").first():
        admin = User(username="admin")
        admin.set_password("admin123")  # Change this in production!
        db.add(admin)
        db.commit()
    db.close() 