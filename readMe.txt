Comment l’utiliser

Setup :

Crée un compte Binance, génère API Key/Secret (lecture seule suffit).
Ajoute-les au .env avec ton webhook.
Installe les dépendances.


Lancement Auto : python bot.py → Attends minuit UTC pour analyser et envoyer les signaux.
Lancement Manuel : python bot.py --manual → Analyse immédiate et envoie les signaux.
Webhook : Le signal arrive en JSON {content: "<html_report>"}. Assure-toi que ton webhook Make.com parse le HTML.