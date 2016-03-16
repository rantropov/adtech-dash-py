
CREATE TABLE received_ads (
    received_at TEXT NOT NULL,
    true_label INTEGER NOT NULL,
    predicted_label INTEGER NOT NULL,
    feature_values TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS
received_at_index ON received_ads ( received_at );

-- CREATE VIEW on received_ads to compute metrics