from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(X_train_resampled, y_train_resampled,
              validation_data=(X_val, y_val),
              epochs=200, batch_size=64, class_weight=class_weights,
              callbacks=[early_stopping, reduce_lr, model_checkpoint])
    return model
