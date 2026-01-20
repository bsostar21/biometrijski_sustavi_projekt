# Zaštita od prijevare otiska prsta 

Ovaj repozitorij sadrži projekt izrađen u sklopu kolegija **Biometrijski sustavi**.
Cilj projekta je detekcija spoofing napada na sustave za prepoznavanje otiska prsta
primjenom konvolucijskih neuronskih mreža (CNN) i transfer learninga.

## Metodologija
- Binarna klasifikacija: live vs spoof
- CNN model s transfer learningom (MobileNetV2, ImageNet)
- Proširenje skupa podataka (data augmentation)
- Fine-tuning modela
- Evaluacija pomoću konfuzijske matrice i standardnih metrika

## Skupovi podataka
- IEEE: Biometric Spoofing Fingerprint Dataset
- Kaggle: Sokoto Coventry Fingerprint Dataset (SOCOFing)

## Izradili
- Luka Videc
- Borna Šoštar
