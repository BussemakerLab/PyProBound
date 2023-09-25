{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :member-order: bysource
   :undoc-members:

   .. automethod:: __init__
   
   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
   {% for item in methods %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: {{ _('Non-Inherited Members') }}
