DROP DATABASE IF EXISTS subleva;
CREATE DATABASE IF NOT EXISTS subleva;

USE subleva;

CREATE TABLE IF NOT EXISTS users
(
    name     VARCHAR(255) NOT NULL,
    dob      DATETIME,
    userType VARCHAR(255),
    id       INT,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS countries
(
    name     VARCHAR(255) UNIQUE  NOT NULL,
    location DECIMAL(7, 5) UNIQUE NOT NULL,
    id       INT,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS volunteers
(
    name            VARCHAR(255) NOT NULL,
    languagesSpoken VARCHAR(255),
    supervisorID    INT,
    id              INT,
    PRIMARY KEY (id),
    FOREIGN KEY (supervisorID) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS appointments
(
    migrantID     INT NOT NULL,
    volunteerID   INT NOT NULL,
    date          DATETIME,
    appointmentID INT NOT NULL,
    PRIMARY KEY (appointmentID),
    foreign key (migrantID) REFERENCES users (id),
    foreign key (volunteerID) REFERENCES volunteers (id)
);

CREATE TABLE IF NOT EXISTS posts
(
    postID      INT,
    postContent VARCHAR(1500),
    createdAt   DATETIME,
    displayName VARCHAR(70),
    migrantID   INT NOT NULL,
    PRIMARY KEY (postID),
    FOREIGN KEY (migrantID) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS attendeeEvents
(
    attendeeID INT,
    eventID    INT,
    primary key (atteNdeeID, eventID),
    FOREIGN KEY (attendeeID) REFERENCES users(id),
    FOREIGN KEY (eventID) REFERENCES communityEvent(eventID)
);

CREATE TABLE IF NOT EXISTS communityEvent
(
    date          DATETIME,
    eventID       INT,
    name          VARCHAR(255),
    duration      INT,
    venueCapacity INT
)