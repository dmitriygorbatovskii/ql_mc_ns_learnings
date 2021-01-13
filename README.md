Скачивание и запуск симулятора

Для запуска обучения необходимы симулятор lgsvl, доступный на официальном сайте https://www.lgsvlsimulator.com и на github: https://github.com/lgsvl/simulator.

Python API с обученными весами доступен в этом репозитории. После клонирования репозитория выполните следующие действия:

1.Выполните следующую команду, чтобы установить файлы Python и необходимые зависимости:
pip3 install --user -e .

2.Теперь запустите симулятор (либо двоичный файл .exe, либо из редактора Unity). Симулятор по умолчанию прослушивает подключения на порту 8181 на локальном хосте.

3.Нажмите Open Browserкнопку, чтобы открыть пользовательский интерфейс симулятора.

4.После загрузки карт и транспортных средств по умолчанию перейдите на Simulationsвкладку.

5.Создайте новую симуляцию. Дайте ему имя и отметьте API Only опцию. Щелкните Submit.

6.Запустите random_agent.py 




