#ifndef VECTOR_TESTING
#define VECTOR_TESTING

namespace gpu{
    /**
     * 
     * This class will test important methods of 
     * Vector class. Tests will focus on success of operations
     * on d_vec. The testing will not be rigorous.
     * 
    */
    class VectorTesting{

        public:

            VectorTesting();

            void testVectorConstructor();
            void testCopyConstructor();
            void testDot();
            void testTensor();
            void testDeepCopy();
            void testEqualOperator();
            void testIsEqualOperator();
            void testMultOperator();
            void testMultAssignOperator();
            void testSubAssignOperator();
    };
}

#endif // End of VECTOR_TESTING