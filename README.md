Скачивание и запуск симулятора

Для запуска обучения необходимы симулятор lgsvl, доступный на официальном сайте https://www.lgsvlsimulator.com и на github: https://github.com/lgsvl/simulator.

Python API с обученными весами доступен в этом репозитории. После клонирования репозитория выполните следующие действия:

1.Выполните следующую команду, чтобы установить файлы Python и необходимые зависимости:
pip3 install --user -e .

2.Теперь запустите симулятор (либо двоичный файл .exe, либо из редактора Unity). Симулятор по умолчанию прослушивает подключения на порту 8181 на локальном хосте.

3.Нажмите Open Browser кнопку, чтобы открыть пользовательский интерфейс симулятора.

4.После загрузки карт и транспортных средств по умолчанию перейдите на Simulations вкладку.

5.Создайте новую симуляцию. Дайте ему имя и отметьте API Only опцию. Щелкните Submit.

6.Запустите random_agent.py со следующими параметрами:

  **##--weather - погода**
  
  **##--time - время**
  
  **--alpha - шаг обучения**
  
  **--gma - обесценивающий фактор**
  
  **--epoch - количество циклов**
  
  **--agent - тип агента (--hepl для просмотре всех вариантов)**
  
  **##--npc**
  
  **--train - True/False - обучение новой модели / демонстрация заранее обученной** 

Readme - условия, погода, NPC, агент, среда, алгоритм, начальные параметры в алгоритме, как запускать код. как самостоятельно тренировать агента (со своими параметрами), тестировать, запустить демонстрацию. все этапы, функции подробно комментировать, графики с tensorboard




